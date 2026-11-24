"""Outbound policy of a MicroVM: which destinations it may reach, and credentials the platform adds.

A MicroVM's outbound policy (`egress_policy` in the MicroVMs API) is set when the MicroVM is
created and cannot change afterwards. A MicroVM created from a checkpoint inherits the policy of
the MicroVM the checkpoint came from, so a template's policy is set when the template is built.

- `default_action` applies to every destination no rule matches: `deny` (the API's default) or `allow`.
- `rules`: up to 50 destinations, each an exact hostname, a wildcard (`*.example.com`, which does not
  match `example.com` itself) or an exact IPv4 address.
- `credentials`: up to 16. The platform sets the `Authorization` header of HTTPS requests to the
  credential's host (and path prefix) to the secret; the secret is stored outside the MicroVM, and
  the MicroVMs API never returns it. Credentials can only be set on a MicroVM created from an image.

Outbound traffic of a MicroVM with a policy goes through a platform proxy, set in the
`HTTP_PROXY` and `HTTPS_PROXY` environment variables: only HTTP and HTTPS from clients that use
those variables reach a destination. For a host with a credential, the platform presents a
certificate from its own certificate authority, which the container trusts through `SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS` and similar variables. See docs/reference.md.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import urlsplit

DefaultAction = Literal["allow", "deny"]
CredentialScheme = Literal["bearer", "raw"]

MAX_RULES = 50
MAX_CREDENTIALS = 16
MAX_SECRET_LENGTH = 8192

# Pulling an image from DigitalOcean Container Registry fetches a registry token from the DigitalOcean API.
DOCR_HOST = "registry.digitalocean.com"
DOCR_AUTH_HOST = "api.digitalocean.com"

_LABEL = r"(?!-)[a-z0-9-]{1,63}(?<!-)"
_HOSTNAME = re.compile(rf"^{_LABEL}(\.{_LABEL})+$")


def _is_ipv4(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return True


def _check_rule(rule: str) -> str:
    rule = rule.strip().lower()
    if _is_ipv4(rule):
        return rule
    host = rule[2:] if rule.startswith("*.") else rule
    if not _HOSTNAME.match(host):
        raise ValueError(
            f"egress rule {rule!r} is not an exact hostname, a '*.' wildcard or an IPv4 address "
            "(no scheme, port, path or CIDR range)"
        )
    return rule


def _matches(rule: str, host: str) -> bool:
    return host.endswith(rule[1:]) if rule.startswith("*.") else host == rule


@dataclass(frozen=True)
class OutboundCredential:
    """A secret the platform adds as the `Authorization` header of HTTPS requests to `host`.

    `scheme="bearer"` sends `Bearer <secret>`, `"raw"` sends the secret as the whole header value.
    With `path_prefix`, only requests whose path starts with it get the credential. Build one with
    `EgressPolicy.with_credential(url, secret=...)`.
    """

    host: str
    secret: str = field(repr=False)
    path_prefix: str | None = None
    scheme: CredentialScheme = "bearer"

    def __post_init__(self) -> None:
        host = self.host.strip().lower()
        if host.startswith("*.") or _is_ipv4(host) or not _HOSTNAME.match(host):
            raise ValueError(f"a credential needs an exact hostname, got {self.host!r}")
        object.__setattr__(self, "host", host)
        if self.path_prefix is not None:
            if not self.path_prefix.startswith("/"):
                raise ValueError(f"a credential's path_prefix must start with '/', got {self.path_prefix!r}")
            prefix = self.path_prefix.rstrip("/")
            object.__setattr__(self, "path_prefix", prefix or None)
        if self.scheme not in ("bearer", "raw"):
            raise ValueError(f"a credential's scheme is 'bearer' or 'raw', got {self.scheme!r}")
        if not self.secret:
            raise ValueError(f"the credential for {host} has an empty secret")
        if len(self.secret) > MAX_SECRET_LENGTH:
            raise ValueError(f"the credential for {host} is longer than {MAX_SECRET_LENGTH} characters")

    @classmethod
    def from_url(cls, url: str, *, secret: str, scheme: CredentialScheme = "bearer") -> OutboundCredential:
        """A credential for an `https://` URL: its host, and its path as the prefix."""
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.hostname:
            raise ValueError(f"a credential URL must be https:// with a hostname, got {url!r}")
        if parts.port is not None or parts.query or parts.fragment or parts.username:
            raise ValueError(f"a credential URL takes a host and a path only, got {url!r}")
        return cls(host=parts.hostname, secret=secret, path_prefix=parts.path or None, scheme=scheme)

    def to_api(self) -> dict[str, Any]:
        body: dict[str, Any] = {"host": self.host, "scheme": self.scheme, "secret": self.secret}
        if self.path_prefix:
            body["path_prefix"] = self.path_prefix
        return body


@dataclass(frozen=True)
class EgressPolicy:
    """A MicroVM's outbound policy (see the module docstring).

    Build one with `allow(*hosts)`, `deny_all()` or `allow_all()`, then add credentials with
    `with_credential`. Pass it as `egress=` to `MicroVMLifecycle.get_or_create` or `build_template`,
    or set `MicroVMConfig.egress` (`DO_MICROVM_EGRESS`, `DO_MICROVM_EGRESS_CREDENTIALS`).
    """

    default_action: DefaultAction = "deny"
    rules: tuple[str, ...] = ()
    credentials: tuple[OutboundCredential, ...] = ()

    def __post_init__(self) -> None:
        if self.default_action not in ("allow", "deny"):
            raise ValueError(f"default_action is 'allow' or 'deny', got {self.default_action!r}")
        rules = tuple(dict.fromkeys(_check_rule(rule) for rule in self.rules))
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "credentials", tuple(self.credentials))
        if len(rules) > MAX_RULES:
            raise ValueError(f"an egress policy has at most {MAX_RULES} rules, got {len(rules)}")
        if len(self.credentials) > MAX_CREDENTIALS:
            raise ValueError(f"an egress policy has at most {MAX_CREDENTIALS} credentials, got {len(self.credentials)}")
        hosts = [credential.host for credential in self.credentials]
        duplicates = sorted({host for host in hosts if hosts.count(host) > 1})
        if duplicates:
            raise ValueError(f"an egress policy has at most one credential per host: {duplicates}")
        if self.default_action == "deny":
            blocked = [host for host in hosts if host not in rules]
            if blocked:
                raise ValueError(f"a credential's host must be allowed by an exact rule: {blocked}")

    @classmethod
    def allow(cls, *hosts: str) -> EgressPolicy:
        """Allow these destinations and deny every other one."""
        return cls(default_action="deny", rules=hosts)

    @classmethod
    def deny_all(cls) -> EgressPolicy:
        """Deny every outbound connection."""
        return cls(default_action="deny")

    @classmethod
    def allow_all(cls) -> EgressPolicy:
        """Allow every destination; traffic still goes through the platform proxy, so credentials apply."""
        return cls(default_action="allow")

    @classmethod
    def parse(cls, spec: str) -> EgressPolicy | None:
        """The policy a `DO_MICROVM_EGRESS` value describes: `none`, `*`, or comma-separated hosts.

        An empty value means no policy (None).
        """
        spec = spec.strip()
        if not spec:
            return None
        if spec.lower() == "none":
            return cls.deny_all()
        if spec == "*":
            return cls.allow_all()
        return cls.allow(*(item.strip() for item in spec.split(",") if item.strip()))

    def with_credential(self, url: str, *, secret: str, scheme: CredentialScheme = "bearer") -> EgressPolicy:
        """This policy plus a credential for an `https://` URL (host and path prefix).

        Under `default_action="deny"` the URL's host is added to the rules when no exact rule names it.
        """
        credential = OutboundCredential.from_url(url, secret=secret, scheme=scheme)
        rules = self.rules
        if self.default_action == "deny" and credential.host not in rules:
            rules = (*rules, credential.host)
        return replace(self, rules=rules, credentials=(*self.credentials, credential))

    def for_image(self, oci_ref: str) -> EgressPolicy:
        """This policy as sent with `oci_ref`: for a DigitalOcean Container Registry image under
        `default_action="deny"`, `api.digitalocean.com` is added to the rules, since the image pull
        fetches its registry token there."""
        if self.default_action != "deny" or oci_ref.split("/", 1)[0].lower() != DOCR_HOST or self.allows(DOCR_AUTH_HOST):
            return self
        return replace(self, rules=(*self.rules, DOCR_AUTH_HOST))

    def allows(self, host: str) -> bool:
        """Whether a connection to `host` is allowed by this policy's rules or default action."""
        host = host.lower()
        return self.default_action == "allow" or any(_matches(rule, host) for rule in self.rules)

    def to_api(self) -> dict[str, Any]:
        """The `egress_policy` field of a create request."""
        body: dict[str, Any] = {
            "default_action": self.default_action,
            "rules": [{"ip": rule} if _is_ipv4(rule) else {"domain": rule} for rule in self.rules],
        }
        if self.credentials:
            body["credentials"] = [credential.to_api() for credential in self.credentials]
        return body

    def digest(self, key: bytes) -> str:
        """A short hash of the whole policy, for cache keys; secrets enter only through an HMAC keyed with `key`."""
        digest = hashlib.sha256()
        for part in (self.default_action, *self.rules):
            digest.update(part.encode() + b"\0")
        for credential in self.credentials:
            mac = hmac.new(key, credential.secret.encode(), hashlib.sha256).digest()
            for part in (credential.host, credential.path_prefix or "", credential.scheme):
                digest.update(part.encode() + b"\0")
            digest.update(mac)
        return digest.hexdigest()[:12]
