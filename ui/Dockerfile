# CDR UI Dockerfile
FROM node:20-slim

WORKDIR /app

# Install serve globally
RUN npm install -g serve

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY . .

# Build for production
RUN npm run build || npx vite build --mode development

# Expose port
EXPOSE 5173

# Serve the built files with SPA fallback
CMD ["serve", "-s", "dist", "--single", "-l", "5173"]
