"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const supertest_1 = __importDefault(require("supertest"));
const index_1 = __importDefault(require("../index"));
const models_1 = require("../models");
describe('Auth Routes', () => {
    beforeEach(async () => {
        await models_1.User.destroy({ where: {} });
    });
    describe('POST /api/auth/register', () => {
        it('should register a new user', async () => {
            const response = await (0, supertest_1.default)(index_1.default)
                .post('/api/auth/register')
                .send({
                name: 'Test User',
                email: 'test@example.com',
                password: 'password123',
                teamNumber: 1234,
            });
            expect(response.status).toBe(201);
            expect(response.body).toHaveProperty('token');
            expect(response.body.user).toHaveProperty('id');
        });
    });
    describe('POST /api/auth/login', () => {
        it('should login an existing user', async () => {
            await models_1.User.create({
                name: 'Test User',
                email: 'test@example.com',
                password: 'password123',
                teamNumber: 1234,
            });
            const response = await (0, supertest_1.default)(index_1.default)
                .post('/api/auth/login')
                .send({
                email: 'test@example.com',
                password: 'password123',
            });
            expect(response.status).toBe(200);
            expect(response.body).toHaveProperty('token');
        });
    });
});
//# sourceMappingURL=auth.test.js.map