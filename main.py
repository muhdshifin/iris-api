import os
import joblib
import jwt

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends, HTTPException

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from pwdlib import PasswordHash

from schemas import IrisInput


# 1. Create FastAPI
app = FastAPI()


# 2. Load ML Model
model = joblib.load("model.pkl")

iris_names = ["setosa", "versicolor", "virginica"]


# 3. JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "development-only-secret")

ALGORITHM = "HS256"


# 4. Password Hashing
password_hash = PasswordHash.recommended()


# 5. Demo Users
users = {

    "user": {
        "password": password_hash.hash("user123"),
        "role": "user"
    },

    "admin": {
        "password": password_hash.hash("admin123"),
        "role": "admin"
    }

}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# 6. Create Token
def create_token(username, role):

    payload = {

        "sub": username,

        "role": role,

        "exp": datetime.now(timezone.utc) + timedelta(minutes=30)

    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


# 7. Login Endpoint
@app.post("/login")

def login(
    form_data: OAuth2PasswordRequestForm = Depends()
):

    user = users.get(form_data.username)

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    if not password_hash.verify(
        form_data.password,
        user["password"]
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    token = create_token(
        form_data.username,
        user["role"]
    )

    return {

        "access_token": token,

        "token_type": "bearer"

    }


# 8. Authentication
def get_current_user(
    token: str = Depends(oauth2_scheme)
):

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")

        user = users.get(username)

        if user is None:

            raise HTTPException(
                status_code=401,
                detail="User not found"
            )

        return {

            "username": username,

            "role": user["role"]

        }

    except jwt.InvalidTokenError:

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )


# 9. Authorization - Admin Only
def admin_only(
    current_user: dict = Depends(get_current_user)
):

    if current_user["role"] != "admin":

        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return current_user


# 10. Home
@app.get("/")
def home():

    return {"message": "Iris API Running"}


# 11. User Profile
@app.get("/profile")
def profile(
    current_user: dict = Depends(get_current_user)
):

    return current_user


# 12. Admin Endpoint
@app.get("/admin")
def admin(
    current_user: dict = Depends(admin_only)
):

    return {"message": "Welcome Admin"}


# 13. ML Prediction
@app.post("/predict")
def predict(
    data: IrisInput,

    current_user: dict = Depends(get_current_user)
):

    input_data = [[

        data.sepal_length,

        data.sepal_width,

        data.petal_length,

        data.petal_width

    ]]

    prediction = model.predict(input_data)

    result = iris_names[int(prediction[0])]

    return {

        "prediction": result,

        "message": "Prediction successful"

    }