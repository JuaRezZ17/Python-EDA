from pathlib import Path
import pandas as pd

class CSVLoader:
    def __init__(self, path, usecols=None, parse_dates=None, dtype=None, required_cols=None):
        self.path = Path(path)
        self.usecols = usecols
        self.parse_dates = parse_dates
        self.dtype = dtype
        self.required_cols = required_cols or []

    def load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.path}")
        try:
            df = pd.read_csv(self.path, usecols=self.usecols,
                             parse_dates=self.parse_dates, dtype=self.dtype)
        except ValueError as e:
            raise ValueError(f"Error al leer CSV ({self.path}): {e}")
        if df.empty:
            raise ValueError(f"CSV vacío: {self.path}")
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas {missing} en {self.path}")
        return df
