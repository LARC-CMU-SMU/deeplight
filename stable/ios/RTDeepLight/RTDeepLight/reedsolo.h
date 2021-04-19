#ifndef __REEDSOLO_H__
#define __REEDSOLO_H__

void rscode_init(int sym_bits, int blolck_size, int ecc_size);
int rscode_encode_bits(int *raw_data, int raw_length, int *enc_data, int *enc_length);
int rscode_decode_bits(int *enc_data, int enc_length, int *dec_data, int *dec_length);
int rscode_encode_symbols(int *raw_data, int raw_length, int *enc_data, int *enc_length);
int rscode_decode_symbols(int *enc_data, int enc_length, int *dec_data, int *dec_length);

#endif /* __REEDSOLO_H__ */
