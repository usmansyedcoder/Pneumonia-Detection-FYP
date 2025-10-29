from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
import os

def generate_keys():
    os.makedirs("keys", exist_ok=True)

    # Generate private key
    private_key = ec.generate_private_key(ec.SECP256R1())
    with open("keys/private_key.pem", "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        )

    # Generate public key
    public_key = private_key.public_key()
    with open("keys/public_key.pem", "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )

    print("✅ ECC key pair generated successfully in 'keys/' folder")

if __name__ == "__main__":
    generate_keys()
