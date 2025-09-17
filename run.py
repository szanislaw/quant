import sys
from pyngrok import ngrok, conf
import streamlit.web.cli as stcli

# ----------------------------
# NGROK CONFIG (explicit)
# ----------------------------
ngrok.set_auth_token("32Y3GYRyDSdwYo7INZhEpmeJVTD_BcLj1NdhuzLpfoqfEf8c")  # add here directly if needed

public_url = ngrok.connect(8501)
print("🌍 Public URL:", public_url)

# ----------------------------
# STREAMLIT STARTUP
# ----------------------------
sys.argv = ["streamlit", "run", "ngrok.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
sys.exit(stcli.main())
