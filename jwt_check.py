import jwt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
JWT_EXP_DELTA_SECONDS = int(os.getenv("JWT_EXP_DELTA_SECONDS"))

# Function to create a JWT
def create_jwt_token(username):
    payload = {
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

# Function to decode a JWT
def decode_jwt_token(token):
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded
    except jwt.ExpiredSignatureError:
        return "Token expired"
    except jwt.InvalidTokenError:
        return "Invalid token"

# Test the JWT implementation
if __name__ == "__main__":
    print("=== JWT Testing ===")
    
    # Create a token
    username = "testuser"
    token = create_jwt_token(username)
    print(f"Generated JWT Token:\n{token}\n")
    
    # Decode the token
    print("Decoded Payload:")
    print(decode_jwt_token(token))
