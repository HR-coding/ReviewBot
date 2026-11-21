from pydantic import BaseModel, Field

# Pull Request
class Repository(BaseModel):
    full_name: str = Field(min_length=1)

class PRHead(BaseModel):
    sha: str = Field(min_length=1)

class User(BaseModel):
    login: str


class PullRequest(BaseModel):
    head: PRHead
    title: str
    user: User  # GitHub sends the author as pull_request.user.login (there is no "author" key)

class PREvent(BaseModel):
    action: str = Field(min_length=1)
    number: int
    repository: Repository
    pull_request: PullRequest
