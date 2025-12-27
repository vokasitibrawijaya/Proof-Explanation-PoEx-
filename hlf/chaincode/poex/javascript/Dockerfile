FROM node:18-alpine

WORKDIR /usr/src/app

COPY package.json ./
RUN npm install --omit=dev

COPY index.js ./
COPY lib ./lib
COPY server.js ./

ENV CHAINCODE_SERVER_ADDRESS=0.0.0.0:9999

CMD ["node", "server.js"]
