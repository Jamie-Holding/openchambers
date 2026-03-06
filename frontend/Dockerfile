FROM node:20-slim

WORKDIR /app

# Install dependencies (layer cached until package files change)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source (overridden by bind mount in dev)
COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0"]
