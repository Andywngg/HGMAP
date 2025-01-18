'use strict';

const bcrypt = require('bcryptjs');

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    const password = await bcrypt.hash('otisit!!!', 10);

    await queryInterface.bulkInsert('users', [{
      name: 'Admin User',
      email: '1334admin@gmail.com',
      password,
      teamNumber: 1334,
      createdAt: new Date(),
      updatedAt: new Date()
    }]);
  },

  async down(queryInterface, Sequelize) {
    await queryInterface.bulkDelete('users', { email: '1334admin@gmail.com' });
  }
}; 