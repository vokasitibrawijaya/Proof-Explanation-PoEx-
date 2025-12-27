FROM node:18-alpine

WORKDIR /app

# Install curl for healthcheck
RUN apk add --no-cache curl

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy Hardhat project
COPY . .

# Compile contracts
RUN npx hardhat compile

EXPOSE 8545

# Default command runs Hardhat node
CMD ["npx", "hardhat", "node", "--hostname", "0.0.0.0"]
