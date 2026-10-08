class UserFacingError(Exception):
    """An error whose message is safe to show the user verbatim.

    Task 10 adds the first raises (upload validation and the image
    pipeline, caught locally in the chat controller); the app-level
    handler that turns it into an HTTP 200 bubble lands in Task 14.
    """

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message
