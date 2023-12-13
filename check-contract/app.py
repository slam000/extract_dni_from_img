import pandas as pd

# Carga CONTRATOS_DNIS.csv y devuelve el contrato_id en base al dni
def dame_contrato_id(dni):
    """
    Carga CONTRATOS_DNIS.csv y devuelve el contrato_id en base al dni

    Args:
        dmi (str): DNI, NIE o pasaporte.

    Returns:
        str: Contrato_id.
    
    Example csv file:
        sociedad;contrato;interl_comercial;dni;contrato_id
        2000;2300000000000;1000000003;12345678A;2000/2300000000000
    """
    # Cargar el archivo csv
    df = pd.read_csv('check-contract/CONTRATOS_DNIS.csv', sep=';')
    # Buscar el contrato_id en base al dni, si no lo encuentra devuelve None
    contrato_id = None
    if dni in df['dni'].values:
        contrato_id = df[df['dni'] == dni]['contrato_id'].values[0]
    return contrato_id


