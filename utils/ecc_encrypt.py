import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def encrypt_image(image_path, public_key_path, encrypted_path):
    # Load public key
    with open(public_key_path, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    # Generate ephemeral ECC key
    ephemeral_key = ec.generate_private_key(ec.SECP256R1())
    shared_key = ephemeral_key.exchange(ec.ECDH(), public_key)

    # Derive AES key from shared secret
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"ecc_aes"
    ).derive(shared_key)

    # AES-GCM encryption
    iv = os.urandom(12)
    with open(image_path, "rb") as f:
        data = f.read()

    encryptor = Cipher(
        algorithms.AES(derived_key),
        modes.GCM(iv)
    ).encryptor()

    ciphertext = encryptor.update(data) + encryptor.finalize()

    with open(encrypted_path, "wb") as f:
        f.write(iv + encryptor.tag + ciphertext)

    # Return ephemeral public key for decryption
    ephemeral_pub = ephemeral_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return ephemeral_pub
