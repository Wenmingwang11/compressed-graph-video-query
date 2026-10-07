import math

def discretize_attributes(df, discretize_columns):
  for column, func in discretize_columns:
    df[column] = df[column].apply(func)

def discretize_function_1(edges, theta_n_round, d_n_round):
  discretize_attributes(edges, [
    ('theta', lambda x: round(x, theta_n_round)),
    ('d_ratio', lambda x: round(x, d_n_round))
  ])

def discretize_function_2(edges):
  discretize_attributes(edges, [
    ('theta', lambda x: x // (math.pi / 2)),
    ('d_ratio', lambda _: 0)
  ])

def discretize_function_3(edges, theta_n_round):
  discretize_attributes(edges, [
    ('theta', lambda x: round(x, theta_n_round)),
    ('d_ratio', lambda _: 0)
  ])

def discretize_function_4(edges, theta_n_parts, theta_d_parts):
  discretize_attributes(edges, [
    ('theta', lambda x: x // (math.pi / theta_n_parts)),
    ('d_ratio', lambda x: x // (1 / theta_d_parts))
  ])
