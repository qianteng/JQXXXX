# -*- coding: utf-8 -*-
"""
@author: Eric Guo <guoanjie@gmail.com>
@brief: utils for text messaging

"""

from twilio.rest import Client

def _send(body):
	ACCOUNTSID="ACadcb7996ef51f60d31936dda8e0e2d7f"
	AUTHTOKEN="a8924f6949b207054a0195d119642006"
	client = Client(ACCOUNTSID, AUTHTOKEN)

	message = client.messages.create(
		to="+19175586830",
		from_="+16467601473",
		body=body,
	)
