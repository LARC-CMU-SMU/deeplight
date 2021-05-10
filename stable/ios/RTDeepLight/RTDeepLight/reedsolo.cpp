#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

class DivideZeroException : public std::exception
{
    const char *info;
public:
    DivideZeroException(void) {
        info = "Divided by zero exception!";
    }
    DivideZeroException(const char* str) {
        info = str;
    }
    virtual const char* what() const throw()
    {
        return info;
    }
    const char * getInfo() const {  return info; }
};

class ValueException : public std::exception
{
    const char *info;
public:
    ValueException(void) {
        info = "Value exception!";
    }
    ValueException(const char* str) {
        info = str;
    }
    virtual const char* what() const throw()
    {
        return info;
    }
    const char * getInfo() const {  return info; }
};

class ReedSolomonException : public std::exception
{
    const char *info;
public:
    ReedSolomonException(void) {
        info = "RS decode exception!";
    }
    ReedSolomonException(const char* str) {
        info = str;
    }
    virtual const char* what() const throw()
    {
        return info;
    }
    const char * getInfo() const {  return info; }
};


class RSCodec
{
    int sym_size;
    int block_size;
    int ecc_size;
    int field_charac;
    std::vector<int>  gf_exp;
    std::vector<int>  gf_log;
    std::vector<int>  poly_generator;
private:
    int gf_add(int x, int y) {
        return x ^ y;
    }

    int gf_sub(int x, int y) {
        return x ^ y;
    }

    int gf_neg(int x) {
        return x;
    }

    int gf_inverse(int x) {
        int index = field_charac - gf_log[x];
        if(index < 0) index = (int)gf_exp.size() + index;
        return gf_exp[index];
    }

    int gf_mul(int x, int y) {
        if((x == 0) || (y == 0)) return 0;
        return gf_exp[(gf_log[x] + gf_log[y]) % field_charac]; //assured index is positive
    }

    int gf_div(int x, int y){
        if(y == 0)
            throw DivideZeroException();
        if(x == 0)
            return 0;
        int index = (gf_log[x] + field_charac - gf_log[y]) % field_charac;
        if(index < 0) index = (int)gf_exp.size() + index;
        return gf_exp[index];
    }

    int gf_pow(int x, int power) {
        int index = (gf_log[x] * power) % field_charac;
        if(index < 0) index = (int)gf_exp.size() + index;
        return gf_exp[index];
    }

    int gf_mult_noLUT(int x, int y, int prim=0, int field_charac_full=256, bool carryless=true) {
        int r = 0;
        while(y) {
            if(y & 1) {
                if(carryless) r = r ^ x;
                else r = r + x;
            }
            y = y >> 1;
            x = x << 1;
            if((prim > 0) && (x & field_charac_full))
                x = x ^ prim;
        }
        return r;
    }

    //################### GALOIS FIELD POLYNOMIALS MATHS ###################

    std::vector<int> gf_poly_scale(std::vector<int> p, int x) {
        std::vector<int> result(p.size());
        for(int i = 0; i < p.size(); i++) result[i] = gf_mul(p[i], x);
        return result;
    }

    std::vector<int> gf_poly_add(std::vector<int> p, std::vector<int>q) {
        std::vector<int> result(std::max(p.size(), q.size()), 0);
        // int inter_size = std::min(p.size(), q.size());
        for(int i = 0; i < p.size(); i++) {
            result[result.size() - p.size() + i] = p[i];
        }
        for(int i = 0; i < q.size(); i++) {
            result[result.size() - q.size() + i] ^= q[i];
        }
        return result;
    }

    std::vector<int> gf_poly_mul(std::vector<int> p, std::vector<int> q) {
        std::vector<int> result(p.size() + q.size() - 1, 0);
        std::vector<int> lp(p.size());
        for(int i = 0; i < p.size(); i++) lp[i] = gf_log[p[i]];
        for(int j = 0; j < q.size(); j++) {
            int qj = q[j];
            if(qj != 0) {
                int lq = gf_log[qj];
                for(int i = 0; i < p.size(); i++) {
                    if(p[i] != 0) {
                        result[i + j] ^= gf_exp[lp[i] + lq];
                    }
                }
            }
        }
        return result;
    }

    std::vector<int> gf_poly_mul_simple(std::vector<int> p, std::vector<int> q) {
        std::vector<int> result(p.size() + q.size() - 1, 0);
        for(int j = 0; j < q.size(); j++) {
            for(int i = 0; i < p.size(); i++) {
                result[i + j] ^= gf_mul(p[i], q[j]);
            }
        }
        return result;
    }

    std::vector<int> gf_poly_neg(std::vector<int> poly) {
        return poly;
    }

    std::vector<int> gf_poly_div(std::vector<int> dividend, std::vector<int> divisor) {
        std::vector<int> result(dividend.size());
        for(int i = 0; i < dividend.size(); i++) result[i] = dividend[i];
        for(int i = 0; i < dividend.size() - (divisor.size() - 1); i++) {
            int coef = result[i];
            if(coef != 0) {
                for(int j = 1; j < divisor.size(); j++) {
                    if(divisor[j] != 0) result[i + j] ^= gf_mul(divisor[j], coef);
                }
            }
        }
        return result;
    }

    std::vector<int> gf_poly_square(std::vector<int> poly) {
        int length = (int)poly.size();
        std::vector<int> result(2*length - 1, 0);
        for(int i = 0; i < length-1; i++) {
            int p = poly[i];
            int k = 2*i;
            if(p != 0) result[k] = gf_exp[2*gf_log[p]];
        }
        result[2*length-2] = gf_exp[2*gf_log[poly[length-1]]];
        if(result[0] == 0) result[0] = 2*poly[1] - 1;
        return result;
    }

    int gf_poly_eval(std::vector<int> poly, int x) {
        int y = poly[0];
        for(int i = 1; i < poly.size(); i++) y = gf_mul(y, x) ^ poly[i];
        return y;
    }

    std::vector<int> rwh_primes1(int n) {
        std::vector<int> result;
        result.push_back(2);
        int max_loop = n+1;
        for(int i = 3; i < max_loop; i += 2) {
            bool is_prime = true;
            for(int j = 0; j < result.size(); j++) {
                if(i % result[j] == 0) {
                    is_prime = false;
                    break;
                }
            }
            if(is_prime) result.push_back(i);
        }
        return result;
    }

    std::vector<int> find_prime_polys(int generator=2, int c_exp=8, bool fast_primes=false) { //Only support 2
        int root_charac = 2;
        int field_charac = (0x01 << c_exp) - 1;
        int field_charac_next = (0x01 << (c_exp+1)) - 1;

        std::vector<int> prim_candidates;
        if(fast_primes) {
            std::vector<int> tem = rwh_primes1(field_charac_next);
            for(int i = 0; i < tem.size(); i++) if(tem[i] > field_charac) prim_candidates.push_back(tem[i]);
        } else {
            for(int i = field_charac+2; i < field_charac_next; i += root_charac) prim_candidates.push_back(i);
        }

        std::vector<int> correct_primes;
        for(int p = 0; p < prim_candidates.size(); p++) {
            int prim = prim_candidates[p];
            std::vector<int> seen(field_charac+1, 0);
            bool conflict = false;

            int x = 1;
            for(int i = 0; i < field_charac; i++) {
                x = gf_mult_noLUT(x, generator, prim, field_charac+1);

                if((x > field_charac) || (seen[x] == 1)) {
                    conflict = true;
                    break;
                } else {
                    seen[x] = 1;
                }
            }
            if(!conflict) correct_primes.push_back(prim);
        }
        return correct_primes;
    }

    void init_tables(int prim, int generator) {
        int x = 1;
        for(int i = 0; i < field_charac; i++) {
            gf_exp[i] = x;
            gf_log[x] = i;
            x = gf_mult_noLUT(x, generator, prim, field_charac+1);
        }
        for(int i = field_charac; i < field_charac * 2; i++) {
            gf_exp[i] = gf_exp[i - field_charac];
        }
    }

    //################### REED-SOLOMON ENCODING ###################

    std::vector<int> rs_generator_poly(int nsym, int fcr=0, int generator=2) {
        std::vector<int> g(1, 1); //1 element, value of 1
        std::vector<int> tem(2);
        for(int i = 0; i < nsym; i++) {
            tem[0] = 1;
            tem[1] = gf_pow(generator, i+fcr);
            g = gf_poly_mul(g, tem);
        }
        return g;
    }

    //################### REED-SOLOMON DECODING ###################

    std::vector<int> rs_calc_syndromes(std::vector<int> msg, int nsym, int fcr=0, int generator=2) {
        std::vector<int> syndromes(nsym + 1);
        for(int i = 0; i < nsym; i++) syndromes[i+1] = gf_poly_eval(msg, gf_pow(generator, i+fcr));
        syndromes[0] = 0;
        return syndromes;
    }

    std::vector<int> rs_correct_errata(std::vector<int> msg_in, std::vector<int> synd, std::vector<int> err_pos, int fcr=0, int generator=2){
        std::vector<int> msg(msg_in);
        std::vector<int> coef_pos(err_pos.size());
        for(int i = 0; i < err_pos.size(); i++) coef_pos[i] = (int)msg.size() - 1 - err_pos[i];
        std::vector<int> err_loc = rs_find_errata_locator(coef_pos, generator);
        std::vector<int> reverse_synd(synd.size());
        for(int i = 0; i < synd.size(); i++) reverse_synd[i] = synd[synd.size()-1-i];
        std::vector<int> reverse_err = rs_find_error_evaluator(reverse_synd, err_loc, (int)err_loc.size()-1);
        std::vector<int> err_eval(reverse_err.size());
        for(int i = 0; i < err_eval.size(); i++) err_eval[i] = reverse_err[reverse_err.size()-1-i];

        std::vector<int> X(coef_pos.size());
        for(int i = 0; i < coef_pos.size(); i++) {
            int l = field_charac - coef_pos[i];
            X[i] = gf_pow(generator, -l);
        }

        std::vector<int> E(msg.size());
        int Xlength = (int)X.size();
        for(int i = 0; i < X.size(); i++) { //i, Xi in enumerate(X):
            int Xi = X[i];
            int Xi_inv = gf_inverse(Xi);

            std::vector<int> err_loc_prime_tmp;
            for(int j = 0; j < Xlength; j++) {
                if(j != i) {
                    err_loc_prime_tmp.push_back( gf_sub(1, gf_mul(Xi_inv, X[j])) );
                }
            }

            int err_loc_prime = 1;
            for(int coef_index = 0; coef_index < err_loc_prime_tmp.size(); coef_index++) {
                int coef = err_loc_prime_tmp[coef_index];
                err_loc_prime = gf_mul(err_loc_prime, coef);
            }

            int y = gf_poly_eval(reverse_err, Xi_inv);
            y = gf_mul(gf_pow(Xi, 1-fcr), y);
            
            int magnitude = gf_div(y, err_loc_prime);
            E[err_pos[i]] = magnitude;
        }
        msg = gf_poly_add(msg, E);
        return msg;
    }

    std::vector<int> rs_find_error_locator(std::vector<int> synd, int nsym, std::vector<int> erase_loc) {
        std::vector<int> err_loc(erase_loc);
        std::vector<int> old_loc(erase_loc);
        if(erase_loc.size() == 0) {
            err_loc = {1};
            old_loc = {1};
        }
        
        int synd_shift = 0;
        if(synd.size() > nsym) synd_shift = (int)synd.size() - nsym;

        for(int i = 0; i < nsym - erase_loc.size(); i++) {
            int K = (int)erase_loc.size() + i + synd_shift;
            int delta = synd[K];
            for(int j = 1; j < err_loc.size(); j++) {
                int synd_index = K - j;
                if(synd_index < 0) synd_index = (int)synd.size() + synd_index;
                delta ^= gf_mul(err_loc[err_loc.size() - (j+1)], synd[synd_index]);
            }

            old_loc.push_back(0);

            if(delta != 0) {
                if(old_loc.size() > err_loc.size()) {
                    std::vector<int> new_loc = gf_poly_scale(old_loc, delta);
                    old_loc = gf_poly_scale(err_loc, gf_inverse(delta));
                    err_loc = new_loc;
                }
                err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta));

            }
        }
        std::vector<int> result;
        for(int i = 0; i < err_loc.size(); i++) {
            if((err_loc[i] != 0) || (result.size() > 0)) result.push_back(err_loc[i]);
        }
        int errs = (int)result.size() - 1;
        if(errs * 2 > nsym) throw ReedSolomonException("Too many errors to correct");
        return result;
    }

    std::vector<int> rs_find_errata_locator(std::vector<int> e_pos, int generator=2) {
        std::vector<int> e_loc = {1};
        std::vector<int> one = {1};
        std::vector<int> pow_ext = {0, 0};
        for(int i = 0; i < e_pos.size(); i++) {
            pow_ext[0] = gf_pow(generator, e_pos[i]);
            e_loc = gf_poly_mul( e_loc, gf_poly_add(one, pow_ext) );
        }
        return e_loc;
    }

    std::vector<int> rs_find_error_evaluator(std::vector<int> synd, std::vector<int> err_loc, int nsym) {
        std::vector<int> dummy(nsym+2, 0);
        dummy[0] = 1;
        std::vector<int> result = gf_poly_div( gf_poly_mul(synd, err_loc), dummy );
        std::vector<int> remainder(result.end() - (dummy.size() - 1), result.end());
        return remainder;
    }

    std::vector<int> rs_find_errors(std::vector<int> err_loc, int nmess, int generator=2) {
        int errs = (int)err_loc.size() - 1;
        std::vector<int> err_pos;
        for(int i = 0; i < nmess; i++) {
            if(gf_poly_eval(err_loc, gf_pow(generator, i)) == 0) {
                err_pos.push_back(nmess - 1 - i);
            }
        }
        
        if(err_pos.size() != errs) {
            throw ReedSolomonException("Too many (or few) errors found by Chien Search for the errata locator polynomial!");
        }
        return err_pos;
    }

    std::vector<int> rs_forney_syndromes(std::vector<int> synd, std::vector<int> pos, int nmess, int generator=2) {
        std::vector<int> erase_pos_reversed(pos.size());// = [nmess-1-p for p in pos];
        for(int i = 0; i < pos.size(); i++) erase_pos_reversed[i] = nmess - 1 - pos[i];
        
        std::vector<int> fsynd(synd.size()-1);// = list(synd[1:])
        for(int i = 0; i < synd.size()-1; i++) fsynd[i] = synd[i+1];
        for(int i = 0; i < pos.size(); i++) { //i in xrange(len(pos)):
            int x = gf_pow(generator, erase_pos_reversed[i]);
            for(int j = 0; j < fsynd.size()-1; j++) { //j in xrange(len(fsynd) - 1):
                fsynd[j] = gf_mul(fsynd[j], x) ^ fsynd[j + 1];
            }
        }
        return fsynd;
    }
public:
    RSCodec(int sym_bits, int num_syms, int ecc_syms)
    {
        sym_size = sym_bits;
        block_size = num_syms;
        ecc_size = ecc_syms;
        int gf_size = 0x01 << sym_bits; //2 ^ sym_bits
        field_charac = gf_size - 1;
        gf_exp = std::vector<int>(2 * field_charac, 1);
        gf_log = std::vector<int>(field_charac + 1, 0);
        std::vector<int> prims = find_prime_polys(2, sym_size);
        init_tables(prims[0], 2);
        poly_generator = rs_generator_poly(ecc_size, 0, 2);
    }

    std::vector<int> encode_symbols(std::vector<int> msg_in) {
        if((msg_in.size() + ecc_size) > field_charac) throw ValueException();

        std::vector<int> msg_out(msg_in.size() + poly_generator.size() - 1);
        for(int i = 0; i < msg_in.size(); i++) msg_out[i] = msg_in[i];
        for(int i = (int)msg_in.size(); i < msg_out.size(); i++) msg_out[i] = 0;

        std::vector<int> lgen(poly_generator.size());
        for(int i = 0; i < poly_generator.size(); i++) lgen[i] = gf_log[poly_generator[i]];

        for(int i = 0; i < msg_in.size(); i++){
            int coef = msg_out[i];
            if(coef != 0) {
                int lcoef = gf_log[coef];
                for(int j = 1; j < poly_generator.size(); j++) {
                    msg_out[i + j] ^= gf_exp[lcoef + lgen[j]];
                }
            }
        }
        for(int i = 0; i < msg_in.size(); i++) msg_out[i] = msg_in[i];
        return msg_out;
    }

    std::vector<int> encode_bits(std::vector<int> bits) {
        std::vector<int> msg_symbols(bits.size()/sym_size);
        int offset = 0;
        int sym = 0;
        int sym_index = 0;
        for(int i = 0; i < bits.size(); i++) {
            sym = (sym << 1) | bits[i];
            offset++;
            if((offset % sym_size) == 0) {
                msg_symbols[sym_index] = sym;
                offset = 0;
                sym = 0;
                sym_index++;
            }
        }
        std::vector<int> enc_symbols = encode_symbols(msg_symbols);
        std::vector<int> enc_bits(enc_symbols.size()*sym_size);
        for(int i = 0; i < enc_symbols.size(); i++) {
            int sym_bits_upper = i*sym_size+sym_size-1;
            int value = enc_symbols[i];
            for(int j = 0; j < sym_size; j++) {
                int bit_value = value & 0x01;
                value = value >> 1;
                enc_bits[sym_bits_upper - j] = bit_value;
            }
        }
        return enc_bits;
    }

    std::vector<int> decode_symbols(std::vector<int> msg_in) {
        int fcr=0;
        int generator=2;
        std::vector<int> erase_pos; //Dummy, no erasure
        if(msg_in.size() > field_charac) {
            throw ValueException("Message is too long");
        }

        std::vector<int> msg_out(msg_in);
        for(int i = 0; i < erase_pos.size(); i++) {
            msg_out[erase_pos[i]] = 0;
        }
        
        if(erase_pos.size() > ecc_size) throw ReedSolomonException("Too many erasures to correct");
        std::vector<int> synd = rs_calc_syndromes(msg_out, ecc_size, fcr, generator);
        if(*std::max_element(synd.begin(), synd.end()) == 0) {
            return std::vector<int>(msg_out.begin(), msg_out.end() - ecc_size);
        }
        
        std::vector<int> fsynd = rs_forney_syndromes(synd, erase_pos, (int)msg_out.size(), generator);
        std::vector<int> err_loc = rs_find_error_locator(fsynd, ecc_size, erase_pos);
        std::vector<int> reverse_err(err_loc.size());
        for(int i = 0; i < err_loc.size(); i++) reverse_err[i] = err_loc[err_loc.size()-1-i];
        std::vector<int> err_pos = rs_find_errors(reverse_err, (int)msg_out.size(), generator);
        if(err_pos.size() == 0) {
            throw ReedSolomonException("Could not locate error");
        }

        for(int i = 0; i < err_pos.size(); i++) erase_pos.push_back(err_pos[i]);
        msg_out = rs_correct_errata(msg_out, synd, erase_pos, fcr, generator);
        synd = rs_calc_syndromes(msg_out, ecc_size, fcr, generator);
        if(*std::max_element(synd.begin(), synd.end()) > 0) {
            throw ReedSolomonException("Could not correct message");
        }
        return std::vector<int>(msg_out.begin(), msg_out.end() - ecc_size);
    }

    std::vector<int> decode_bits(std::vector<int> bits) {
        std::vector<int> msg_symbols(bits.size()/sym_size);
        int offset = 0;
        int sym = 0;
        int sym_index = 0;
        for(int i = 0; i < bits.size(); i++) {
            sym = (sym << 1) | bits[i];
            offset++;
            if((offset % sym_size) == 0) {
                msg_symbols[sym_index] = sym;
                offset = 0;
                sym = 0;
                sym_index++;
            }
        }
        std::vector<int> dec_symbols = decode_symbols(msg_symbols);
        std::vector<int> dec_bits(dec_symbols.size()*sym_size);
        for(int i = 0; i < dec_symbols.size(); i++) {
            int sym_bits_upper = i*sym_size+sym_size-1;
            int value = dec_symbols[i];
            for(int j = 0; j < sym_size; j++) {
                int bit_value = value & 0x01;
                value = value >> 1;
                dec_bits[sym_bits_upper - j] = bit_value;
            }
        }
        return dec_bits;
    }
};

RSCodec codec(5, 20, 10);

void rscode_init(int sym_bits, int blolck_size, int ecc_size) {
    codec = RSCodec(sym_bits, blolck_size, ecc_size);
}

int rscode_encode_bits(int *raw_data, int raw_length, int *enc_data, int *enc_length) {
    std::vector<int> msg(raw_data, raw_data + raw_length);
    std::vector<int> enc;
    try {
        enc = codec.encode_bits(msg);
    } catch(std::exception& e) {
//        std::cout<<"Exception RSCODE: "<<e.what()<<std::endl;
        return -1;
    }
    std::memcpy((void*)enc_data, (void*)enc.data(), enc.size()*sizeof(int));
    *enc_length = (int)enc.size();
    return 0;
}

int rscode_decode_bits(int *enc_data, int enc_length, int *dec_data, int *dec_length) {
    std::vector<int> msg(enc_data, enc_data + enc_length);
    std::vector<int> dec;
    try {
        dec = codec.decode_bits(msg);
    } catch(std::exception& e) {
//        std::cout<<"Exception RSCODE: "<<e.what()<<std::endl;
        return -2;
    }
    std::memcpy((void*)dec_data, (void*)dec.data(), dec.size()*sizeof(int));
    *dec_length = (int)dec.size();
    return 0;
}

int rscode_encode_symbols(int *raw_data, int raw_length, int *enc_data, int *enc_length) {
    std::vector<int> msg(raw_data, raw_data + raw_length);
    std::vector<int> enc;
    try {
        enc = codec.encode_symbols(msg);
    } catch(std::exception& e) {
//        std::cout<<"Exception RSCODE: "<<e.what()<<std::endl;
        return -1;
    }
    std::memcpy((void*)enc_data, (void*)enc.data(), enc.size()*sizeof(int));
    *enc_length = (int)enc.size();
    return 0;
}

int rscode_decode_symbols(int *enc_data, int enc_length, int *dec_data, int *dec_length) {
    std::vector<int> msg(enc_data, enc_data + enc_length);
    std::vector<int> dec;
    try {
        dec = codec.decode_symbols(msg);
    } catch(std::exception& e) {
//        std::cout<<"Exception RSCODE: "<<e.what()<<std::endl;
        return -2;
    }
    std::memcpy((void*)dec_data, (void*)dec.data(), dec.size()*sizeof(int));
    *dec_length = (int)dec.size();
    return 0;
}
