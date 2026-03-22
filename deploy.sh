expect -c '
set timeout -1
spawn ssh -o StrictHostKeyChecking=no root@173.242.55.249
expect "password:"
send "Tsvjg8LzRSXD\r"
expect "#"
send "cd /root/swing-bot && git pull\r"
expect "#"
send "source venv/bin/activate && pip install pybit python-dotenv\r"
expect "#"
send "cp ob-bot.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable ob-bot && systemctl restart ob-bot\r"
expect "#"
send "systemctl status ob-bot --no-pager\r"
expect "#"
send "exit\r"
expect eof
'
