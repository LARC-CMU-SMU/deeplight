from codec import *
import numpy as np

sym_bits = 9 #Number of bit per symbol
block_len = 20 #Number of symbols in a block (we transmit data in a block)
ratio = 0.5 #Ratio of data symbol in a block. The rest are error correction code
msg_len = int(ratio*block_len) #Number of data symbols
codec = Codec(sym_bits=sym_bits, block_len=block_len, code_ratio=ratio)

#Generate test data
msg = np.random.randint(0,2,90, dtype=np.uint8)
#Encode test data
enc = codec.encode(msg)

print(enc)
# #Test random error
# error_ratio = 0.05
# error_indexes = np.random.randint(0, len(enc), int(error_ratio*len(enc)))
# for idx in error_indexes:
#     if enc[idx] == 0:
#         enc[idx] = 1
#     else:
#         enc[idx] = 0
# # Test burst error
offset = 5
for i in range(9):
    if enc[offset+i] == 0: 
        enc[offset+i] = 1
    else:
        enc[offset+i] = 0
print(enc)

#Decode the noisy data
try:
    dec = codec.decode(enc)
except Exception as error:
    print(error)
print("")
print(msg)
print(dec)

#Compare original data and decoded data
for i in range(len(msg)):
    if msg[i] != dec[i]:
        print("Different")
        break