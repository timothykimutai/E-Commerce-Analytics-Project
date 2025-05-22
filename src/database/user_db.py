from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.models.user import User, UserInDB, get_password_hash

# Database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(db: SessionLocal, username: str) -> Optional[UserInDB]:
    user = db.query(UserModel).filter(UserModel.username == username).first()
    if user:
        return UserInDB(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            disabled=user.disabled,
            is_admin=user.is_admin,
            hashed_password=user.hashed_password
        )
    return None

def get_user_by_email(db: SessionLocal, email: str) -> Optional[UserInDB]:
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if user:
        return UserInDB(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            disabled=user.disabled,
            is_admin=user.is_admin,
            hashed_password=user.hashed_password
        )
    return None

def create_user(db: SessionLocal, user: User, password: str) -> UserInDB:
    hashed_password = get_password_hash(password)
    db_user = UserModel(
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        hashed_password=hashed_password,
        disabled=user.disabled,
        is_admin=user.is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return UserInDB(
        id=db_user.id,
        email=db_user.email,
        username=db_user.username,
        full_name=db_user.full_name,
        disabled=db_user.disabled,
        is_admin=db_user.is_admin,
        hashed_password=db_user.hashed_password
    )

def get_all_users(db: SessionLocal) -> List[User]:
    users = db.query(UserModel).all()
    return [
        User(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            disabled=user.disabled,
            is_admin=user.is_admin
        )
        for user in users
    ]

def update_user(db: SessionLocal, user_id: int, user_data: dict) -> Optional[User]:
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if user:
        for key, value in user_data.items():
            if hasattr(user, key):
                setattr(user, key, value)
        db.commit()
        db.refresh(user)
        return User(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            disabled=user.disabled,
            is_admin=user.is_admin
        )
    return None

def delete_user(db: SessionLocal, user_id: int) -> bool:
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        return True
    return False 