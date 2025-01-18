"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.up = up;
exports.down = down;
const sequelize_1 = require("sequelize");
async function up(queryInterface) {
    await queryInterface.createTable('teams', {
        id: {
            type: sequelize_1.DataTypes.INTEGER,
            autoIncrement: true,
            primaryKey: true,
        },
        teamNumber: {
            type: sequelize_1.DataTypes.INTEGER,
            allowNull: false,
            unique: true,
        },
        autoScoreCoral: {
            type: sequelize_1.DataTypes.BOOLEAN,
            defaultValue: false,
        },
        autoScoreAlgae: {
            type: sequelize_1.DataTypes.BOOLEAN,
            defaultValue: false,
        },
        autoStartingPosition: {
            type: sequelize_1.DataTypes.STRING,
        },
        teleopCoralCapability: {
            type: sequelize_1.DataTypes.INTEGER,
            defaultValue: 0,
        },
        teleopAlgaeCapability: {
            type: sequelize_1.DataTypes.INTEGER,
            defaultValue: 0,
        },
        teleopDealgifying: {
            type: sequelize_1.DataTypes.BOOLEAN,
            defaultValue: false,
        },
        teleopCoralLevels: {
            type: sequelize_1.DataTypes.ARRAY(sequelize_1.DataTypes.STRING),
            defaultValue: [],
        },
        endgameDeepPerformance: {
            type: sequelize_1.DataTypes.INTEGER,
            defaultValue: 0,
        },
        endgameShallowPerformance: {
            type: sequelize_1.DataTypes.INTEGER,
            defaultValue: 0,
        },
        endgameClimbing: {
            type: sequelize_1.DataTypes.BOOLEAN,
            defaultValue: false,
        },
        robotWidth: {
            type: sequelize_1.DataTypes.FLOAT,
        },
        robotLength: {
            type: sequelize_1.DataTypes.FLOAT,
        },
        robotHeight: {
            type: sequelize_1.DataTypes.FLOAT,
        },
        robotWeight: {
            type: sequelize_1.DataTypes.FLOAT,
        },
        drivetrainType: {
            type: sequelize_1.DataTypes.STRING,
        },
        notes: {
            type: sequelize_1.DataTypes.TEXT,
        },
        imageUrl: {
            type: sequelize_1.DataTypes.STRING,
        },
        createdAt: {
            type: sequelize_1.DataTypes.DATE,
            allowNull: false,
        },
        updatedAt: {
            type: sequelize_1.DataTypes.DATE,
            allowNull: false,
        },
    });
}
async function down(queryInterface) {
    await queryInterface.dropTable('teams');
}
//# sourceMappingURL=20240111000002-create-teams.js.map