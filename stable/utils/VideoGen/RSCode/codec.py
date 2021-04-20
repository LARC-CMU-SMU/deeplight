import numpy as np
import math
import reedsolo

class Codec(object):
    def __init__(self, sym_bits=5, block_len=20, code_ratio=0.8):
        self.sym_bits = sym_bits
        self.block_len = block_len
        self.code_ratio = code_ratio
        self.msg_len = int(block_len*code_ratio)
        self.ecc_len = self.block_len - self.msg_len
        self.prim = reedsolo.find_prime_polys(c_exp=self.sym_bits)
        self.codec = reedsolo.RSCodec(nsym=self.ecc_len, prim=self.prim[0], nsize=self.block_len, c_exp=self.sym_bits)
    def encode(self,input_data):
        '''
        input_data: an array of binary values (data) with a length of self.msg_len*self.sym_size
        '''
        # if self.msg_len*self.sym_bits != len(input_data):
        #     print("Input data length is different from total number of bits in a message for encoding. Please apply data padding to generate correct number of bits")
        #     return None
        total_sym_padded = math.ceil(len(input_data)/self.sym_bits)
        symbols = bytearray(total_sym_padded)
        tem_data = np.concatenate((input_data, np.zeros(total_sym_padded*self.sym_bits-len(input_data), dtype=np.uint8))).reshape((total_sym_padded, self.sym_bits))
        for i in range(len(tem_data)):
            tem = 0
            for k in range(self.sym_bits):
                tem = (tem << 1) | (tem_data[i][k]&0x01)
            symbols[i] = tem
        enc = self.codec.encode(symbols)
        result = np.zeros(len(enc)*self.sym_bits, dtype=np.uint8)
        for i in range(len(enc)):
            for k in range(self.sym_bits):
                result[i*self.sym_bits+self.sym_bits-k-1] = enc[i]&0x01
                enc[i] = enc[i] >> 1
        return result

    def decode(self, input_data):
        '''
        input_data: an array of binary values (data) with a length of self.blk_len*self.sym_size
        '''
        # if self.block_len*self.sym_bits != len(input_data):
        #     print("Input data length is different from total number of bits in a message for decoding")
        #     return None
        symbols = bytearray(int(len(input_data)/self.sym_bits))
        tem_data = input_data.reshape((len(symbols), self.sym_bits))
        for i in range(len(tem_data)):
            tem = 0
            for k in range(self.sym_bits):
                tem = (tem << 1) | (tem_data[i][k]&0x01)
            symbols[i] = tem
        dec = self.codec.decode(symbols)
        result = np.zeros(len(dec)*self.sym_bits, dtype=np.uint8)
        for i in range(len(dec)):
            for k in range(self.sym_bits):
                result[i*self.sym_bits+self.sym_bits-k-1] = dec[i]&0x01
                dec[i] = dec[i] >> 1
        return result

