from trulens_eval import Tru
from dotenv import load_dotenv
import os

load_dotenv()

tru = Tru(database_url=os.getenv('DATABASE_URL'))
# tru.reset_database()
tru.run_dashboard()