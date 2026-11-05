# AlphaForge frontend — Next.js 15.5.28 / React 19 dashboard (port 3001)
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# NEXT_PUBLIC_* are inlined into the browser bundle at BUILD time: pass them
# as build args (`--build-arg NEXT_PUBLIC_API_URL=https://...`); setting them
# on the running container changes nothing. No default: an image built
# without an API URL calls no API (the dashboard shows its recorded snapshot)
# instead of sending every visitor's browser to their own localhost.
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_SITE_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_SITE_URL=$NEXT_PUBLIC_SITE_URL
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.mjs ./next.config.mjs
EXPOSE 3001
CMD ["npm", "run", "start"]
