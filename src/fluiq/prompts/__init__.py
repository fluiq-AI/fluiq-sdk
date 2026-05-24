from __future__ import annotations


class Prompt:
    """A prompt template fetched from the Fluiq dashboard.

    Attributes
    ----------
    slug        : str   — the prompt's unique identifier
    name        : str   — human-readable display name
    template    : str   — raw template string with ``{variable}`` placeholders
    model       : str | None — preferred model pinned in the dashboard, if any
    variables   : list[str]  — declared variable names
    version     : int   — deployed version number
    environment : str   — environment this snapshot was fetched from
    """

    def __init__(self, data: dict) -> None:
        self.slug        = data["slug"]
        self.name        = data.get("name", "")
        self.template    = data["template"]
        self.model       = data.get("model")
        self.variables   = data.get("variables") or []
        self.version     = data.get("version", 1)
        self.environment = data.get("environment", "production")

    def render(self, **kwargs: str) -> str:
        """Render the template by substituting ``{variable}`` placeholders.

        Raises ``KeyError`` when a required variable is missing.
        """
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        return (
            f"<Prompt slug={self.slug!r} version={self.version}"
            f" env={self.environment!r}>"
        )


__all__ = ["Prompt"]
