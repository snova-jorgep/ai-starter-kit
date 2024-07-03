import os
import sqlite3
import sys

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.llms.sambanova import SambaStudio, Sambaverse
from langchain_core.prompts import PromptTemplate

from .fastCoE import SambaStudioFastCoE

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
DB_PATH = os.path.join(kit_dir, 'data', 'financial.db')
OUTPUT_PATH = os.path.join(kit_dir, 'data/letters')
MAX_USERS = 100

load_dotenv(os.path.join(repo_dir, '.env'))


def get_user_data(db_path, max_users=None):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the query to get user data along with account type
    query = """
    SELECT customers.CustomerName, customers.Age, accounts.AccountName
    FROM customers
    JOIN accounts ON customers.AccountID = accounts.AccountID
    """

    # Add the limit to the query if max_users is provided
    if max_users is not None:
        query += f' LIMIT {max_users}'

    for row in cursor.execute(query):
        user_name, user_age, account_type = row
        yield (user_name, user_age, account_type)

    # Close the connection
    conn.close()


def create_letters(db_path=DB_PATH, output_path=OUTPUT_PATH, max_users=MAX_USERS):
    os.makedirs(output_path, exist_ok=True)

    llm = SambaStudioFastCoE(
        max_tokens=2048,
        model='llama3-8b',
    )

    prompt = """
    Egy nagyon fejlett nyelvi modell vagy.
    A feladatod egy személyre szabott levél megírása magyar nyelven a bankunk egyik ügyfelének.
    A levél az alábbi alapadatok alapján kerüljön megírásra:
    Ügyfél neve: {customer_name}
    Ügyfél életkora: {customer_age} év
    Bankszámla típusa: {account_type}
    A levél célja az, információ az ügyfelet arról, hogy a számlája aktív, és hogy bármilyen ezzel kapcsolatos kérdéssel hozzánk fordulhat. A levél stílusa hivatalos legyen, és olyan stílus elemeket és nyelvezetet tartalmazzon, ami egy hasonló életkorú személy esetén általános.
    Egy fiatal (18-25 éves közötti) ügyfél esetén a nyelvezet professzionálisnak tűnjön, de egy kicsit barátságosabb hangvételű legyen. Egy középkorú (26-50 év közötti) ügyfél esetén találjuk meg az egyensúlyt a professzionalizmus és a meleg hangvétel között.
    Egy idősebb (51 évesnél idősebb) ügyfél esetén a nyelvezet tiszteletreméltó és megerősítő legyen.
    A levelet az Acme bank küldi.
    A választ a levéllel kezdd.
    """

    prompt_template = PromptTemplate.from_template(prompt)

    for customer in get_user_data(db_path, max_users=max_users):
        try:
            letter = llm.invoke(
                prompt_template.format(customer_name=customer[0], customer_age=customer[1], account_type=customer[2])
            )
            with open(os.path.join(output_path, f'{customer[0]}_letter.txt'), 'w') as file:
                file.write(letter)

            print(f'letter for customer {customer[0]} created')

        except:
            print(f'letter for customer {customer[0]} not created')
