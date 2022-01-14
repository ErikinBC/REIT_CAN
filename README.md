This repo contains the code necessary to generate all the figures and analysis used for this post: [Shorting the Canadian housing market with REITs](http://www.erikdrysdale.com/reit_can_short). To replicate the analysis:

1. Configure the conda environment: `conda env create -f env_reit.yml`
2. Run the three scripts: `source pipeline.sh`

Note that results of this code are based on data that was available as of June 25th, 2021, and will naturally change when run in the future due to new market data and house price information. 

Additional specs:

1. python 3.9.5
2. ubuntu 18.04.4 LTS
3. conda 4.10.1
