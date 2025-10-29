from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def decrypt_image(encrypted_path, private_key_path, ephemeral_pub_bytes, decrypted_path):
    # Load private key
    with open(private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    # Load ephemeral public key
    ephemeral_pub = serialization.load_pem_public_key(ephemeral_pub_bytes)

    # Derive shared key
    shared_key = private_key.exchange(ec.ECDH(), ephemeral_pub)
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"ecc_aes"
    ).derive(shared_key)

    # Read encrypted file
    with open(encrypted_path, "rb") as f:
        file_data = f.read()

    iv, tag, ciphertext = file_data[:12], file_data[12:28], file_data[28:]

    # AES-GCM decryption
    decryptor = Cipher(
        algorithms.AES(derived_key),
        modes.GCM(iv, tag)
    ).decryptor()

    plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    with open(decrypted_path, "wb") as f:
        f.write(plaintext)
