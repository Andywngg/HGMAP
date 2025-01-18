"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.up = up;
exports.down = down;
const bcryptjs_1 = __importDefault(require("bcryptjs"));
async function up(queryInterface) {
    const password = await bcryptjs_1.default.hash('password123', 10);
    await queryInterface.bulkInsert('users', [
        {
            name: 'Demo User',
            email: 'demo@example.com',
            password,
            teamNumber: 1234,
            createdAt: new Date(),
            updatedAt: new Date(),
        },
    ]);
    await queryInterface.bulkInsert('teams', [
        {
            teamNumber: 1234,
            autoScoreCoral: true,
            autoScoreAlgae: false,
            autoStartingPosition: 'Left',
            teleopCoralCapability: 4,
            teleopAlgaeCapability: 3,
            teleopDealgifying: true,
            teleopCoralLevels: ['low', 'mid'],
            endgameDeepPerformance: 5,
            endgameShallowPerformance: 4,
            endgameClimbing: true,
            robotWidth: 30,
            robotLength: 32,
            robotHeight: 24,
            robotWeight: 120,
            drivetrainType: 'Swerve',
            notes: 'Demo team data',
            createdAt: new Date(),
            updatedAt: new Date(),
        },
    ]);
}
async function down(queryInterface) {
    await queryInterface.bulkDelete('users', {});
    await queryInterface.bulkDelete('teams', {});
}
//# sourceMappingURL=20240111000001-demo-data.js.map