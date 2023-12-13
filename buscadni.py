import re

s = """
CHIERNANDODEMAGALIANES4
CRDOBA
PP CORDOBA

ZJ4CORDOBA
4 I CRDOBA
52 ED

MIGUEL7FRANCISCA
IDESPCBH109527830427096M
5702022M3201174ESP9
TEJEROXMUNOZXXFERNANDO
"""
match = re.search(r'\d{8}[A-Za-z]', s)

if match:
    print("DNI encontrado:", match.group())
else:
    print("No se encontró DNI.")