# Forward Proxy Server
Client -> http/https proxy server -> remote websites
```Bash
# Compile nginx with ngx_http_proxy_connect_module
sudo apt update
sudo apt install -y nginx
/usr/sbin/nginx -v

wget http://nginx.org/download/nginx-1.22.1.tar.gz
tar -xf nginx-1.22.1.tar.gz

git clone https://github.com/chobits/ngx_http_proxy_connect_module.git

cd nginx-1.22.1/
patch -p1 < ../ngx_http_proxy_connect_module/patch/proxy_connect_rewrite_102101.patch
./configure --prefix=/usr --add-module=../ngx_http_proxy_connect_module
make
sudo make install

# systemd
systemctl status nginx
vim /lib/systemd/system/nginx.service
ExecStart=/usr/sbin/nginx -c /etc/nginx/nginx.conf -g 'daemon on; master_process on;'
ExecReload=/usr/sbin/nginx -c /etc/nginx/nginx.conf -g 'daemon on; master_process on;' -s reload

# Modify /etc/nginx/nginx.conf
# Comment out ssl directives
# ssl_protocols
# ssl_prefer_server_ciphers

# Add forward proxy server
        server {
            listen 8585;

            # dns resolver used by forward proxying
            resolver                       8.8.8.8;

            # forward proxy for CONNECT requests
            proxy_connect;
            proxy_connect_allow            443;
            proxy_connect_connect_timeout  10s;
            proxy_connect_data_timeout     10s;
            location / {
                proxy_pass $scheme://$http_host$request_uri;
            }
        }

systemctl stop nginx
systemctl start nginx
tail -f /var/log/nginx/access.log
```
<br/>

Client side:<br/>
```Bash
export http_proxy=http://IP:8585
export https_proxy=http://IP:8585
```
