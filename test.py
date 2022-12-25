import sys
import time
import requests
from datetime import datetime
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header


def func():
    S = "aaaa"
    C = [3, 4, 5, 6]
    left = 0
    right = 1
    res = 0
    while left < len(S) and right < len(S):
        if S[left] == S[right]:
            if C[left] >= C[right]:
                res += C[right]
                right = left + 1
            else:
                res += C[left]
                left = right + 1
        else:
            left += 1
            right += 1
    return res


print(func())
