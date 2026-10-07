class UserFacingError(Exception):
    """An error whose message is safe to show the user verbatim.

    Not raised anywhere yet — Task 10 adds the upload-validation raises;
    the app-level handler that turns it into an HTTP 200 bubble lands in
    Task 14.
    """

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
