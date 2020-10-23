For ready to go notebook demonstrating analysis use this link: 
https://mybinder.org/v2/gh/Chrisebeling/nba_player_categorization/40c288c7a714f6ae0b3a72da72e630e52a025132?filepath=Player%20Categorization.ipynb

## Analysis
- Clustering analysis (K Means) of NBA statistics since 1983
- Two separate models are run: career and single season performance
- Average stats across the time period for each player are used as the factors, with equal weighting
- Stats included: pts, rbs, ast, blk, stl, tov, fg3_pct, fg2_pct, ft_pct, fg3a, fg2a, fta
- Clustering algorithm is written from scratch but should be equivalent to algorithms found in common packages
- Clustering algorithm has been run and outputs saved, so the clusters are now static each time the notebook is run
