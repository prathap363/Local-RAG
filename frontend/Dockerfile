# Lightweight web server image for static frontend assets.
FROM nginx:1.27-alpine

# Replace default nginx config and copy static files.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/index.html
COPY app.js /usr/share/nginx/html/app.js
COPY styles.css /usr/share/nginx/html/styles.css
