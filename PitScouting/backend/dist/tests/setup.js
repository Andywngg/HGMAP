"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const models_1 = require("../models");
beforeAll(async () => {
    await models_1.sequelize.sync({ force: true });
});
afterAll(async () => {
    await models_1.sequelize.close();
});
//# sourceMappingURL=setup.js.map