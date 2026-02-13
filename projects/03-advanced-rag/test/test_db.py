import pytest
from db_api import KnowledgeDatabase, Session

"""
使用 pytest 框架编写的数据库单元测试脚本
它的核心目的是独立地验证数据库操作逻辑（CRUD）是否正确，而不依赖于 API 或其他上层服务
"""

# 把 “创建会话→自动回滚→销毁会话” 的逻辑抽成 fixture；
# 所有测试函数只需通过参数 session 调用，无需重复写代码（复用性）。
@pytest.fixture
def session():
    # 1. 创建一个新的数据库会话
    with Session() as session:
        # 2. 执行测试用例前的准备工作（可选）
        #    这里可以添加清理数据等操作，但当前代码没有
        yield session  # 3. 将创建好的 session 对象提供给测试函数使用
        # 4. 执行测试用例后的清理工作
        session.rollback()  # 回滚在测试中对数据库所做的所有更改


# 插入数据功能
def test_insert_knowledge_database(session):
    # 步骤 1: 创建一个新的模型实例（模拟要插入的数据）
    new_record = KnowledgeDatabase(title="test", category="category")
    # 步骤 2: 将实例添加到会话中
    session.add(new_record)
    # 步骤 3: 提交会话，将数据写入数据库
    session.commit()
    # 步骤 4: 验证 - 从数据库中查询刚刚插入的数据
    record = session.query(KnowledgeDatabase).filter_by(title="test").first()
    # 步骤 5: 使用断言（assert）检查结果是否符合预期
    assert record is not None    # 断言：查询结果不应为空
    assert record.title == "test"  # 断言：标题正确
    assert record.category == "category"  # 断言：类别正确
    print("插入知识库成功")


# 查询数据功能
def test_query_knowledge_database(session):
    # 步骤 1: 先向数据库中插入一条测试数据（准备测试环境）
    session.add(KnowledgeDatabase(title="test", category="category"))
    session.commit()

    # 步骤 2: 执行查询操作
    records = session.query(KnowledgeDatabase).filter_by(title="test").all()

    # 步骤 3: 断言结果
    assert len(records) > 0  # 断言：至少查询到一条记录
    assert records[0].title == "test"  # 断言：查询到的记录标题正确
    print("查询知识库成功")


def test_delete_knowledge_database(session):
    # 步骤 1: 先插入一条数据，用于后续删除
    record_to_delete = KnowledgeDatabase(title="test", category="category")
    session.add(record_to_delete)
    session.commit()

    # 步骤 2: 执行删除操作
    records_to_delete = session.query(KnowledgeDatabase).filter_by(title="test").all()
    for record in records_to_delete:
        session.delete(record)
        session.commit()

    # 步骤 3: 验证 - 查询被删除的数据，应该为空
    records = session.query(KnowledgeDatabase).filter_by(title="test").all()
    assert len(records) == 0
    print("删除知识库成功")
