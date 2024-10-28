# Launch a startup script
```Bash
cat <<EOF > ~/Library/LaunchAgents/com.user.jupyter.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
   <key>Label</key>
   <string>com.user.jupyter</string>
   <key>ProgramArguments</key>
   <array><string>/Users/bytedance/bin/jupyter.sh</string></array>
   <key>RunAtLoad</key>
   <true/>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.user.jupyter.plist
launchctl start ~/Library/LaunchAgents/com.user.jupyter.plist
```

