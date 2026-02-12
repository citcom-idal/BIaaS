class ExternalAPIError(Exception):
    """Exception for handling errors related to external API calls."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class PlannerError(Exception):
    """Exception for handling errors related to the visualization planner."""

    def __init__(self, raw_content: str) -> None:
        super().__init__(f"Planner Error: {raw_content}")


class PlannerJSONError(Exception):
    """Exception for handling JSON parsing errors from the visualization planner."""

    def __init__(self, raw_content: str) -> None:
        super().__init__(f"Planner JSON Error: {raw_content}")

        self.raw_content = raw_content
