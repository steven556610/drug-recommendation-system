from fastapi.middleware.cors import CORSMiddleware

def add_cors_middleware(app):
    """
    Applies security policies to the FastAPI instance for web resource access.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Permits rapid integration with external frontends
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )