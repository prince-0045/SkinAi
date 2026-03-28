# Stage 1: Build the React app
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --legacy-peer-deps

# Copy source code
COPY . .

# Build the production bundle
# The backend URL is injected at build time via ARG/ENV
ARG VITE_API_URL=https://api.dermaura.tech
ENV VITE_API_URL=$VITE_API_URL

# vite.config.js uses envDir: '../' so from /app the env dir is /
# Write .env.production there so Vite finds VITE_API_URL at build time
RUN printf 'VITE_API_URL=%s\nVITE_GOOGLE_CLIENT_ID=144232855654-v2bucqk1f3679d06gedppcj1sd3moqnd.apps.googleusercontent.com\n' "$VITE_API_URL" > ../.env.production

RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy built files to nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config to handle React Router
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]