# Multi-stage build: compile the Vite app, then serve the static bundle with nginx.
FROM node:20-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
# Baked in at build time -- Vite inlines env vars starting with VITE_ into the bundle.
ARG VITE_API_BASE_URL=http://localhost:8080
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# nginx-unprivileged: runs as a non-root user and listens on 8080 by default,
# so this works under the Helm chart's hardened securityContext
# (runAsNonRoot, readOnlyRootFilesystem) without extra capabilities.
FROM nginxinc/nginx-unprivileged:1.27-alpine AS production
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
