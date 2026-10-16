"""Endpoint auth for exposing the HTTP server beyond the Docker network.

Off unless MCP_PUBLIC_URL is set, and until then requests that came through a
reverse proxy are refused. When on, the server accepts two kinds of
bearer token:

- MCP_INTERNAL_TOKEN: a static token for internal callers such as n8n. Full
  tool access.
- OAuth tokens issued to one pre-registered client (MCP_OAUTH_CLIENT_ID /
  MCP_OAUTH_CLIENT_SECRET), for Claude.ai custom connectors. Open client
  registration is off, and destructive or server-local tools are hidden.

Issued OAuth tokens live in memory, so Claude.ai users reconnect after a
restart.
"""

import base64
import binascii
import hmac
import os
import re
from typing import Optional
from urllib.parse import parse_qs, unquote, urlencode

from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken, MultiAuth, TokenVerifier
from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import InvalidRedirectUriError, OAuthClientInformationFull
from pydantic import AnyUrl
from starlette.datastructures import Headers
from starlette.middleware import Middleware as ASGIMiddleware
from starlette.responses import JSONResponse

CLAUDE_CALLBACK = "https://claude.ai/api/mcp/auth_callback"
# Claude Code uses a loopback redirect on a random port
LOOPBACK_CALLBACK = re.compile(r"^http://(localhost|127\.0\.0\.1)(:\d+)?/callback$")

INTERNAL_CLAIM = "basecamp_internal"

# Hidden from OAuth (Claude.ai) tokens, along with every trash_* and delete_* tool.
# create_attachment reads a path on this server, so it could upload any file in
# the container, including the Basecamp OAuth token.
OAUTH_HIDDEN_TOOLS = {
    "create_attachment",
    "get_webhooks", "create_webhook",
    "update_project_access",
    "create_project", "update_project", "archive_project", "unarchive_project",
    "update_my_profile", "update_my_preferences",
}


def _hidden_from_oauth(name: str) -> bool:
    return name.startswith(("trash_", "delete_")) or name in OAUTH_HIDDEN_TOOLS


class ClaudeClient(OAuthClientInformationFull):
    """Pre-registered client that accepts Claude.ai and Claude Code redirects."""

    def validate_redirect_uri(self, redirect_uri: Optional[AnyUrl]) -> AnyUrl:
        if redirect_uri is None:
            return AnyUrl(CLAUDE_CALLBACK)
        value = str(redirect_uri)
        if value == CLAUDE_CALLBACK or LOOPBACK_CALLBACK.match(value):
            return redirect_uri
        raise InvalidRedirectUriError(f"Redirect URI '{value}' not allowed")


class StaticClientOAuthProvider(InMemoryOAuthProvider):
    """OAuth server with one fixed client and no dynamic registration."""

    def __init__(self, *, base_url: str, client_id: str, client_secret: str):
        super().__init__(
            base_url=base_url,
            client_registration_options=ClientRegistrationOptions(enabled=False),
            revocation_options=RevocationOptions(enabled=True),
        )
        self.clients[client_id] = ClaudeClient(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=[AnyUrl(CLAUDE_CALLBACK)],
            token_endpoint_auth_method="client_secret_post",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope="mcp offline_access",
        )

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        raise ValueError("Client registration is disabled")


class InternalTokenVerifier(TokenVerifier):
    """Accepts one static bearer token for callers on the internal network."""

    def __init__(self, token: str):
        super().__init__()
        self._token = token.encode()

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        if not hmac.compare_digest(token.encode(), self._token):
            return None
        return AccessToken(token=token, client_id="internal", scopes=[], claims={INTERNAL_CLAIM: True})


def _is_internal_caller() -> bool:
    token = get_access_token()
    return bool(token and token.claims.get(INTERNAL_CLAIM) is True)


class OAuthToolFilter(Middleware):
    """Hide destructive and server-local tools from anyone but internal callers."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        if _is_internal_caller():
            return tools
        return [t for t in tools if not _hidden_from_oauth(t.name)]

    async def on_call_tool(self, context, call_next):
        if not _is_internal_caller() and _hidden_from_oauth(context.message.name):
            raise ToolError(f"Tool '{context.message.name}' is not available on this connection")
        return await call_next(context)


class BasicToPostTokenAuth:
    """Rewrite HTTP Basic client auth on /token into form fields.

    The client is registered as client_secret_post, and the MCP SDK checks one
    method per client. This lets the token endpoint accept either method.
    """

    def __init__(self, app, token_path: str = "/token"):
        self.app = app
        self.token_path = token_path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != self.token_path:
            await self.app(scope, receive, send)
            return

        auth = Headers(scope=scope).get("authorization", "")
        if not auth.lower().startswith("basic "):
            await self.app(scope, receive, send)
            return

        try:
            client_id, client_secret = base64.b64decode(auth[6:]).decode("utf-8").split(":", 1)
        except (ValueError, UnicodeDecodeError, binascii.Error):
            await self.app(scope, receive, send)
            return

        body = b""
        more = True
        while more:
            message = await receive()
            body += message.get("body", b"")
            more = message.get("more_body", False)

        form = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        form.setdefault("client_id", [unquote(client_id)])
        form.setdefault("client_secret", [unquote(client_secret)])
        new_body = urlencode(form, doseq=True).encode("utf-8")

        headers = [
            (k, v) for k, v in scope["headers"]
            if k.lower() not in (b"authorization", b"content-length")
        ]
        headers.append((b"content-length", str(len(new_body)).encode()))
        scope = dict(scope, headers=headers)

        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {"type": "http.request", "body": new_body, "more_body": False}

        await self.app(scope, replay, send)


class RejectProxiedRequests:
    """With auth off, refuse anything that came through a reverse proxy.

    Caddy routes a public hostname here whether or not auth is configured. A
    proxied request carries X-Forwarded-For; direct calls on the Docker network
    (n8n) don't. This keeps the server internal-only until MCP_PUBLIC_URL is set.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and "x-forwarded-for" in Headers(scope=scope):
            await JSONResponse({"error": "Remote access is not enabled on this server"}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def build_auth():
    """Return the server auth provider, or None when MCP_PUBLIC_URL is unset."""
    public_url = os.getenv("MCP_PUBLIC_URL", "").strip().rstrip("/")
    if not public_url:
        return None

    required = ("MCP_OAUTH_CLIENT_ID", "MCP_OAUTH_CLIENT_SECRET", "MCP_INTERNAL_TOKEN")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(f"MCP_PUBLIC_URL is set but {', '.join(missing)} missing; refusing to start without auth")

    provider = StaticClientOAuthProvider(
        base_url=public_url,
        client_id=os.environ["MCP_OAUTH_CLIENT_ID"].strip(),
        client_secret=os.environ["MCP_OAUTH_CLIENT_SECRET"].strip(),
    )
    return MultiAuth(server=provider, verifiers=[InternalTokenVerifier(os.environ["MCP_INTERNAL_TOKEN"].strip())])


def asgi_middleware(auth_enabled: bool) -> list:
    if not auth_enabled:
        return [ASGIMiddleware(RejectProxiedRequests)]
    return [ASGIMiddleware(BasicToPostTokenAuth)]
