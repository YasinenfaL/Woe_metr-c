import pandas as pd
import numpy as np

data = {
    'Education': ['University', 'High School', 'University', 'Bachelor', 'Bachelor', 'High School'],
    'CreditStatus': [1, 0, 1, 1, 0, 0]  # 1: Approval, 0: Denial
}
df = pd.DataFrame(data)


def woe_iv(dataframe, target, variable):
    df = dataframe[[variable, target]]
    df['n'] = 1
    df_group = df.groupby([variable, target], as_index=False).count()
    df_pivot = df_group.pivot(index=variable, columns=target, values='n').fillna(0)
    df_pivot.columns = ['Denial', 'Approval']

    df_pivot['Total'] = df_pivot['Denial'] + df_pivot['Approval']
    df_pivot['DenialPerc'] = df_pivot['Denial'] / df_pivot['Denial'].sum()
    df_pivot['ApprovalPerc'] = df_pivot['Approval'] / df_pivot['Approval'].sum()

    df_pivot['WOE'] = np.log(df_pivot['ApprovalPerc'] / df_pivot['DenialPerc'])
    df_pivot['IV'] = (df_pivot['ApprovalPerc'] - df_pivot['DenialPerc']) * df_pivot['WOE']

    return df_pivot[['WOE', 'IV']]


woe_iv_df = woe_iv(df, 'CreditStatus', 'Education')
print(woe_iv_df)

# Lıbrary feature_engine
import pandas as pd
from feature_engine.encoding import WoEEncoder

data = {
    'Education': ['University', 'High School', 'University', 'Bachelor', 'Bachelor', 'High School'],
    'CreditStatus': [1, 0, 1, 1, 0, 0]  # 1: Approval, 0: Denial
}
df = pd.DataFrame(data)

encoder = WoEEncoder(variables=['Education'])

df_encoded = encoder.fit_transform(df[['Education']], df['CreditStatus'])

print(df_encoded)
