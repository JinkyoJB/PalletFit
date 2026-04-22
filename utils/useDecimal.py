from decimal import Decimal


def set2Decimal(value, number_of_decimals=2):
    number_of_decimals = getLimitNumberOfDecimals(number_of_decimals)

    return Decimal(value).quantize(number_of_decimals)

def getLimitNumberOfDecimals(number_of_decimals):
    return Decimal('1.{}'.format('0' * number_of_decimals))


def cross_decimal(v1, v2):
    """
    Decimal 객체를 포함하는 두 벡터의 외적을 계산합니다.
    """
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]
