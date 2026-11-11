"""Flask routes for signing out of the HTTP Basic login."""

SIGNED_OUT_PAGE = '<p style="font-family:sans-serif">Signed out. <a href="/">Sign in again</a></p>'


def add_logout_route(server) -> None:
    """GET /logout answers 401 with a fresh challenge, so the browser forgets the saved login (close its prompt to stay signed out)."""

    @server.route("/logout")
    def logout():
        return SIGNED_OUT_PAGE, 401, {"WWW-Authenticate": 'Basic realm="Crude Oil Risk Manager"'}
