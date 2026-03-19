# Bots as Infrastructure: A Large-Scale Study of Benign and Malicious Uses on Telegram

This repository contains the code to reproduce the results of the paper published at ICWSM'26: A Large-Scale Study of Telegram Bots (Tsuchiya et al, 2026). 
You are expected to have at least 25 MB of disk space and to execute two scripts in less than 10 minutes on a consumer laptop.

Notes: 
- As discussed in the ethical statement of the paper, this repository (publicly available) does not contain any data files.
- The scripts show you more detailed steps than what we described in the paper. 
- However, if you would like to directly execute the scripts, please request data access through the Zenodo link [here](https://zenodo.org/records/17281308) and put the csv files under the `data/` directory.
- In `data/`, we have all 32,000+ bots (including the interaction with them). 
- We have two scripts to reproduce the quantitative results (and the figures) in the paper.
- We confirmed that the scripts ran without any issues on multiple machines.

Please follow the instructions below.

1. Set up the environment
    - Create a Python virtual environment (e.g., `python3 -m venv telegram-bots-artifact`) and activate it (`source telegram-bots-artifact/bin/activate`). (We recommend Python 3.12+.)
   - Install the required packages using `pip install -r requirements.txt`.
   - We only use standard packages (pandas, numpy, sklearn, matplotlib, seaborn) for easy reproduction, so you might be able to run the scripts without creating a virtual environment.

2. Create directories for the outputs (if they do not exist)
   - `mkdir data`
   - `mkdir csv`
   - `mkdir figs`

3. Reproduce the results
    - Run `python3 produce_results.py` to generate the results (in standard output and `csv/`).
    - Run `python3 produce_figures.py` to generate the figures (in `figs/`).

4. Verify the results
   - Check the script outputs and figures and compare them with the results in the paper.

## Scripts

- `produce_results.py`: Code to produce the results in the paper.
- `produce_figures.py`: Code to produce the figures in the paper.

## Data
`data/bot_data.csv`
- The main bot data (32,000+ bots)
- Note that this data does not contain messages that mention the bots. You cannot reproduce the same labels from the LLMs without the messages.
- Columns:
    - id_hash (str): SHA-256 hash (the first 8 characters) of the bot ID
    - name_hash (str): SHA-256 hash (the first 8 characters) of the bot username
    - description (str): Bot description. NaN means an empty description.
    - command_list (list): List of commands supported by the bot
    - start_response (str): Response message when we send the /start command
    - help_response (str): Response message when we send the /help command

`data/bot_labels.csv`
- Bot labels (categories, languages)
- Columns:
    - id_hash (str): SHA-256 hash (the first 8 characters) of the bot ID
    - category_gpt-4o-mini (str): Label produced by GPT-4o-mini
    - category_gpt-4o (str): Label produced by GPT-4o
    - description_language (str): Language of the bot description
    - message_most_frequent_language (str): Most frequent language in messages that mention the bot

`data/bot_channel_info.csv`
- Channel information for each bot
- Columns:
    - id_hash (str): SHA-256 hash (the first 8 characters) of the bot ID
    - first_timestamp (int): Unix timestamp of the first message that mentioned the bot
    - last_timestamp (int): Unix timestamp of the last message that mentioned the bot
    - num_msg (int): Total number of messages that mentioned the bot
    - channels (list): List of channels that mentioned the bot
    - density (float): Density for the channel network for the bot
    - avg_degree (float): Average degree for the channel network for the bot

`data/bot_translation.csv`
- English translations of descriptions and command lists
- Columns:
    - id_hash (str): SHA-256 hash (the first 8 characters) of the bot ID
    - description_en (str): English translation of the bot description
    - command_list_en (list): English translation of the bot command list

`bot_annotated_samples.csv`
- Human and LLM annotations for 100 random bots
- Columns:
    - id_hash (str): SHA-256 hash (the first 8 characters) of the bot ID
    - category_annotator1 (str): Category label by annotator 1
    - category_annotator2 (str): Category label by annotator 2
    - category_human (str): Final label by humans (after discussion between two annotators)
    - category_gpt-4o (str): Category label by GPT-4o
    - category_gpt-4o-mini (str): Category label by GPT-4o-mini
