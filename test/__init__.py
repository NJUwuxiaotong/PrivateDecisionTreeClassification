import math


def u_cf_of_50(total_num, error_num):
    coff = 0.6925
    confidence_level = 0.5
    value = (error_num + confidence_level + coff * coff / 2 +
             math.sqrt(coff * coff * ((error_num + confidence_level) *
                       (1 - (error_num + confidence_level)/total_num) +
                       coff * coff/4))) / (total_num + coff*coff)
    return value


print("<0, 6>: %s" % u_cf_of_50(6, 0))
print("<0, 1>: %s" % u_cf_of_50(1, 0))
print("<1, 2>: %s" % u_cf_of_50(2, 1))
print("<0, 9>: %s" % u_cf_of_50(9, 0))
print("<1, 16>: %s" % u_cf_of_50(16, 1))
print("<1, 168>: %s" % u_cf_of_50(168, 1))
