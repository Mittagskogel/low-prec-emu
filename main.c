#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <fenv.h>  // Rounding modes

#include <omp.h>



// Must be in C for type-punning to be defined
union fpenum{
    uint16_t it;
    _Float16 fp;
};

_Float16 chop(float a) {
    // FP 16
    // 0    c   8   4
    // SEEEEEMMMMMMMMMMM
    
    union {
        float fp32;
        uint32_t conv;
        _Float16 fp16;
    } chop;
    chop.fp32 = a;
    
    //printf("%x", chop.conv);

    // float fields
    uint32_t m = chop.conv & 0x007fe000;
    uint32_t e = chop.conv & 0x7f800000;
    uint32_t s = chop.conv & 0x80000000;

    e >>= 23;
    
    // Inf or NaN
    if (e == 0x000000ff) {
        e = 0x0000001f;
    }
    // Overflow
    else if (e > 142) {
        e = 0x1f;
        m = 0;
    }
    // Underflow
    else if (e < 102) {
        m = 0;
        e = 0;
    }
    // Subnormal numbers
    else if (e < 113) {
        m |= 0x00800000;
        // UB if shifting more than 23
        uint32_t shift = 113 - e;
        m >>= shift;

        uint32_t round_comp = 0x00001000;
        round_comp <<= shift;
        
        // round ties to even
        if ((m & 0x00003fff) == 0x00003000) {
            m += 0x00002000;
        }
        // round to nearest
        else if ((chop.conv & (round_comp - 1)) > 0 && (m & 0x00001000)) {
            m += 0x00002000;
        }
        
        // mantissa overflow?
        if (m & 0x00800000) {
            e = 1;
        } else {
            e = 0;
        }
        
        m &= 0x007fe000;
    }
    // Normal conversion
    else {
        // rounding ties to even
        // add 1 to mantissa
        if ((chop.conv & 0x00003fff) == 0x00003000) {
            m += 0x00002000;
        }
        // round to nearest
        // add 1 to mantissa
        else if ((chop.conv & 0x00001fff) > 0x00001000) {
            m += 0x00002000;
        }

        // mantissa overflow?
        // add 1 to exponent
        if (m & 0x00800000) {
            e += 1;
        }

        m &= 0x007fe000;
        
        e -= 112;
    }
    e &= 0x0000001f;


    
    chop.conv = (s >> 16) | (e << 10) | (m >> 13);

    //printf("->%x %f\n", chop.conv, (float) chop.fp16);
    
    return chop.fp16;
}

#define OP +
#define OP_STR "-"

int main(int argc, char* argv[]) {
    printf("Operation is " OP_STR "\n");
    
    // Get rounding mode
    int rmode = fegetround();

    printf("Rounding mode is ");
    switch (rmode) {
    case (FE_DOWNWARD):
        printf("FE_DOWNWARD");
        break;
    case (FE_TONEAREST):
        printf("FE_TONEAREST");
        break;
    case (FE_TOWARDZERO):
        printf("FE_TOWARDZERO");
        break;
    case (FE_UPWARD):
        printf("FE_UPWARD");
        break;
    default:
        printf("Unknown rounding mode!");
        exit(-1);
    }
    printf(".\n");
    
     
    union fpenum i, j;

#pragma omp parallel for private(i, j)
    for (uint32_t it = 0; it < 0x10000; ++it) {
        i.it = it;
        for (uint32_t jt = 0; jt < 0x10000; ++jt) {
            j.it = jt;
            
            union fpenum fp_hw;
            fp_hw.fp = i.fp OP j.fp;

            
            float a = (float) i.fp;
            float b = (float) j.fp;
            float c = a OP b;
            
            union fpenum fp_chop;
            fp_chop.fp = chop(c);

            
            if (fp_hw.fp != fp_chop.fp) {
                // Lazily compare NaNs
                if ((fp_hw.it & 0x7c00) == 0x7c00 && (fp_chop.it & 0x7c00) == 0x7c00) {
                    if ((fp_hw.it & 0x7e00) == (fp_chop.it & 0x7e00)) {
                        continue;
                    }
                }
                
                union {
                    float fp32;
                    uint32_t conv;
                } chop_dbg;
                chop_dbg.fp32 = c;

#pragma omp critical
                {
                    printf("Chop: %x (%e) -> %x (%e)\n", chop_dbg.conv, c, fp_chop.it, (float) fp_chop.fp);
                    
                    printf("0x%x (%e) " OP_STR "0x%x (%e) = hw: 0x%x (%e) chop: 0x%x (%e)\n",
                           i.it, (float) a, j.it, (float) b,
                           fp_hw.it, (float) fp_hw.fp, fp_chop.it, (float) fp_chop.fp);
                    exit(-1);
                }
            }
        }
    }

    return 0;
}
