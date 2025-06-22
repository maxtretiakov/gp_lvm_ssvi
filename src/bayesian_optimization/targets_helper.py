import pandas as pd
from pathlib import Path

def load_targets(work_dir: Path) -> pd.DataFrame:
    """
    Load the JG067 sequence targets, add PrimerPair, EvaGreen, and PrimerPairReporter,
    and drop duplicates. Fully matches the notebook logic.
    """
    targets = pd.read_csv(work_dir / 'data/JG067 sequence targets.csv', index_col=[0])

    # Construct PrimerPair column
    targets['PrimerPair'] = targets[['FPrimer', 'RPrimer']].agg('-'.join, axis=1)

    # Determine EvaGreen or Probe
    is_evagreen = (
        (targets['-Strand Label'] == "None") & 
        (targets['+Strand Label'] == "None")
    )
    targets.loc[is_evagreen, 'EvaGreen'] = 'EvaGreen'
    targets.loc[~is_evagreen, 'EvaGreen'] = 'Probe'

    # Combine to PrimerPairReporter
    targets['PrimerPairReporter'] = targets[['PrimerPair', 'EvaGreen']].agg('-'.join, axis=1)

    # Keep only the first occurrence for each PrimerPairReporter
    targets = targets.drop_duplicates(subset=['PrimerPairReporter'], keep='first')

    return targets
