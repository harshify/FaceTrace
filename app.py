import streamlit as st
import pandas as pd
import time
from datetime import datetime

ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")

from streamlit_autorefresh import st_autorefresh

count = st_autorefresh(interval=5000, limit=100, key="counter")

st.write("### Today's attendance")


df=pd.read_csv("Attendance/Attendance_" + date + ".csv")

st.dataframe(df.style.highlight_max(axis=0))

